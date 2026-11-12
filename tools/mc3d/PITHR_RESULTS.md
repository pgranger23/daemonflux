# pithr: charged-pion tracking threshold (muon-chain neutrinos), 2026-11-12

Scope: `tools/mc3d/shower.py` `thresholds(e_nu_min, scheme)`, `tools/mc3d/sky3d.py --thresholds {legacy,chain}`,
`tools/mc3d/test_thresholds.py`. Derivation: `runs/pithr/DERIVATION.md`. Default is `legacy`: nothing in production B or mcup changes.

## Bounds (e_nu_min = 0.1 GeV, total energy)
The neutrino of a chain from a parent satisfies E_nu <= (E + p)/2 of the parent (E*_mu + p*_mu = m_parent exactly for pi, K -> mu nu).
Hence pi+-: E + p >= 0.2, i.e. E >= 0.1487 GeV (legacy 0.2342, which bounds only the two-body neutrino; the muon chain reaches
(E_pi+p_pi)/2 = 0.211 GeV at the legacy cut). Kaons: never dropped (m_K > 2 e_nu_min; legacy 0.105 was also below m_K). Muons: as legacy
(exact bound 0.128). Nucleons: Fermi-smeared NNpi threshold 1.045 (legacy 1.174; the backend gives no pion below E ~ 1.7 GeV, so inert).
Antinucleons, Lambda, anti-Lambda: all kept (annihilation; Lambda -> p pi-).

## Bit identity (option off)
Production-B physics, seed 9182736, 1970 primaries, 4 workers: pristine code (`runs/pithr/orig`) vs edited code with `legacy`:
10 t-files, **0 differing arrays** (including m3b_opts), identical totals (s1 5.077380, t1 8.475448). The `chain` run writes
`thr_scheme = chain` into manifest, t-files and the merged total, and appends [-3, 1] to `m3b_opts`; legacy files carry neither.

## Validation (field-free; LEG seed 31000001, CHN seed 31000002; 444 660 primaries each, 6 workers each)
R = chain/legacy, cap 3 (10 deg), measure 1, sum of 4 species, errors from the spread of per-task sums (the sq-based errors understate
the noise by 3.6-4.4, see PREREG addendum).

| band (GeV) | |cos Z| | R_s | R_t | s/t | pre-registered R_s |
|---|---|---|---|---|---|
| 0.10-0.16 | 0.05 | 1.097 +- 0.068 | 0.855 +- 0.144 | 1.283 | 1.047 (z +0.73) |
| 0.10-0.16 | 0.55 | 1.023 +- 0.027 | 0.973 +- 0.038 | 1.051 | 1.027 (z -0.15) |
| 0.10-0.16 | 0.95 | 1.020 +- 0.025 | 1.048 +- 0.033 | 0.974 | 1.019 (z +0.06) |
| 0.16-0.25 | 0.05/0.55/0.95 | 0.880 +- 0.065 / 0.987 +- 0.028 / 1.036 +- 0.029 | | | in [1, 1.047]: consistent within 1.9 sigma |
| 0.25-1 | 0.05/0.55/0.95 | 1.035 +- 0.147 / 0.962 +- 0.029 / 1.034 +- 0.034 | | | 1: |z| <= 1.3 |
| 1-10 | 0.05/0.55/0.95 | 1.009 +- 0.230 / 1.056 +- 0.092 / 1.053 +- 0.078 | | | 1: |z| <= 0.7 |

All pre-registered expectations are met within 2 sigma, but the precision (2.5 percent at |cos Z| 0.55 and 0.95, 6 percent at 0.05) is
far from the 1 percent target: the sizing used the sq errors. The s/t central values (1.28, 1.05, 0.97) have an error of order 0.05-0.15
(the s-t correlation of the same showers was not evaluated for lack of budget) and are consistent with 1 and with the expected
~1.00; the net effect on the table's factor is **not resolved** by this run at better than ~5 percent.
CPU per primary: legacy 7.89 ms, chain 9.36 ms (task meta; user+sys of the whole job 3989 vs 4620 s): chain/legacy = 1.19 in this
field-free low-energy sample (pre-registered <= 1.05 not met); in the production-B bit-id sample (with cutoff) 1.03.

## Tests (tools/mc3d, `timeout`, one process each)
test_closure 4, test_cutoff 19, test_decays 13, test_geometry 12, test_m3b 15, test_m3c 7, test_m3d 11, test_pool 8,
test_sampler 6, test_sky 13, test_upgoing 34, test_thresholds 14 (new): 156 passed, 0 failed.
