# HANDOFF — 3D/low-energy daemonflux extension (migration to Lyon CC-IN2P3)

Context snapshot for resuming this work on `lyon:/sps/lbno/pgranger/`. Written
2026-07-05. Companion to `IMPLEMENTATION_REPORT.md` (full module inventory),
`PAPER_DRAFT.md` (the paper), `TECHNICAL_NOTE.md`, `README.md`, `REVIEW.md`.

## What this project is
A first-principles **3D + low-energy extension of `daemonflux`** (the
muon-calibrated 1D atmospheric-neutrino flux on MCEq), usable **0.1–100 GeV**,
built to be directly comparable to Honda/Bartol *with full uncertainty
propagation*. Everything is implemented from first principles — no reference
flux enters the construction.

Core object: `Φ_3D = Φ_base(E,|cosZ|) × E_off(E,cosZ) × G(E,R_c(θ,φ)) × S(E)`
- `Φ_base`: hybrid base — GSF-primary MCEq below E0=1.7 GeV, muon-calibrated
  daemonflux above (recommended; `base_model="hybrid"`).
- `E_off`: first-principles off-axis 3D-production factor (generator (x_L,θ)
  pion kernel + exact π→μν decay; flavour-independent; NA61-validated).
- `G`: cascade-correct geomagnetic suppression from a full-IGRF back-traced
  cutoff, now **production-cone-averaged** (`cone_cutoff=True`, the default).
- `S`: optional solar force-field factor.

## Location & branch
- Repo (this migration): `/sps/lbno/pgranger/daemonflux`, branch **`3d-extension`**.
- All work is in `tools/mceq3d/`. The only package-wired piece is
  `src/daemonflux/geomagnetic.py`. Fork remote: `git@github.com:pgranger23/daemonflux.git`.
- Report PDFs: `/sps/lbno/pgranger/3D_extension_report/` (PAPER/NOTE/IMPL).
- Working tree was clean at migration except untracked `moments_deliverables/`.
  Latest commits: `fc8adda` (finer Fig 1), `71dd9d9` (referee response),
  `31976d2` (E-W cone fix).

## Environment (rebuild on the cluster — the macOS .venv was NOT copied)
Python 3.13. Create a fresh venv and install:
```
python -m venv .venv && . .venv/bin/activate
pip install -e .              # daemonflux itself
pip install MCEq crflux chromo ppigrf numpy scipy matplotlib pandas hepdata-cli
```
- `src/daemonflux/data/*.pkl` (spline + calibration files, ~249 MB) **were
  copied** — the engine needs them and daemonflux's lazy GitHub download often
  fails behind firewalls. If missing, download the `daemonsplines_*` pickles.
- `chromo` only ships UrQMD-3.4 + SIBYLL-2.3d offline; EPOS/QGSJET/DPMJET need
  network on first use (blocked in the old sandbox — may work on the cluster).
- MCEq quirk: `import importlib.util` before importing MCEq. numpy 2.x:
  `np.trapz` → `np.trapezoid`.

## Caches copied (so you do NOT recompute the expensive back-traces)
- `tools/mceq3d/.cache3d/` — geomagnetic cutoff maps: the fine Fig-1 map
  (31×61, n_scan=40, ~62 min to rebuild), the cone `finemap_rc` (13×25×20),
  per-direction cutoff grids. Keyed by site/date/grid hash.
- `tools/mceq3d/cutoff_cache/`, `tools/mceq3d/na61_k/` (NA61 K± HEPData).
- Honda/Bartol reference tables are cached as `honda_kam.npz` / `bartol_kam.npz`.

## How to run the key things (from tools/mceq3d, `PYTHONPATH=$PWD`)
- Tests: `python -m pytest -q`  (expect **132 passed**).
- Honda validation + Fig 6: `python validate_honda.py --plot`
- Out-of-sample E-W zenith scan: `python validate_ew_zenith.py`
- Systematic band + Fig 12: `python base_comparison.py`
- Fine cutoff sky map (Fig 1): `python make_cutoff_map_fig.py`  (cached; instant
  re-plot, ~62 min first build).
- Build a PDF: `pandoc <doc>.md -o <out>/<doc>.pdf --pdf-engine=xelatex -H
  pdf_header.tex -V mainfont="STIX Two Text" -V monofont="Menlo" -V
  geometry:margin=1in -V colorlinks=true`

## Current state — where things stand
The paper is submittable-after-coherence per the last review. Recent work
(this session):
1. **E-W gap closed the right way, then corrected for honesty.** Fixed a
   rigidity-cap bug (`r_hi` 20→40 GV — near-horizon East cutoff exceeds 20 GV)
   and added the production-cone average of G (`cone_cutoff`, now the `solve()`
   default). **Important:** the E-W MUST be compared at matched zenith. The old
   `validate_honda` compared this work at 75° vs Honda's ~87° horizon bin — a
   mismatch that flattered it to "few %". At MATCHED zenith the delivered E-W
   **overshoots Honda by ~11–14%** at 75°, growing to ~26% at the extreme
   horizon (cone truncates below the limb where the down-going map is clamped —
   documented limitation, §6 v). Correct sign/peak/vanishing; a real
   improvement over the old ~14% deficit, but NOT exact. Reported honestly.
2. **Systematic band recomputed** around the GSF-hybrid central (not raw
   H3a-MCEq): **±14% at 0.5 GeV** (was ±31%), →0 above ~2 GeV crossover.
3. **Finer Fig 1 cutoff map** (31×61, n_scan=40, smooth Gouraud shading).
4. Abstract cut to ~250 words; Futagami (SK E-W) ref added; docs swept.

## Open / pending items
- **E-W residual overshoot — limb fix DONE (2026-07-05), residual remains.**
  Extended the cutoff map continuously across the limb: `finemap_rc` now builds a
  **full-sphere** map (down-going detector cutoff + up-going far-side cutoff via
  the new `farside_cutoff_map` helper), so the near-horizon down-going production
  cone samples the real far-side cutoff below the limb instead of clamping at 89°.
  `full_sphere=True` is the default; `cone_geff` needed no change (its zenith
  clamp auto-widens). This pulled the extreme-horizon overshoot from ≈25–29% to
  ≈20–25% at 87° (0.5 GeV: +29%→+20%), ≈18–21%→≈18% at 81°; moderate zeniths
  (63/69°) and the 75° matched bin unchanged. Limb continuity verified (E 40.0→39.0,
  W 7.8→7.8 GV across 89°→90°). **Residual overshoot remains** from the
  intrinsically sharp horizon E-W contrast + finite cone quadrature — candidate
  next steps: finer cone quadrature near the horizon, up-going *neutrino*-direction
  cone extension (the analogous far-side change), or accepting the residual as a
  documented boundary. NOTE: the up-going far-side back-trace for the full-sphere
  map is slow (~40–80 min for a 13×25×20 hemisphere; use `sbatch` for rebuilds);
  cached per site/date in `.cache3d/finerc_*.npz`.
- **Channel cone + muon bending (2026-07-06, DONE, both default True).** Chased the
  residual: ruled out quadrature (converged <1%), production-point displacement
  (~6%), Honda binning (~few%). Mostly a genuine factorization limit, with a
  muon-decay-channel piece now fixed: (1) channel-weighted cone (`channel_cone`:
  `cone_geff` blends a wider muon-decay cone by MCEq `f_mu`), (2) muon-bending
  charge-signed cone-axis shift (`muon_bending`). Both first-principles, both help
  modestly (numu 87°/0.5: +20%→+15% cumulative; nue undershoot reduced), no
  regression. `muon_bending` triples cone_geff cost (cached). Bulk of the residual
  is still the factorization limit; next real lever is full 3D transport
  (spherical_streaming/cascade prototypes), not another cone tweak. See
  [[ew-validation-status]] memory.
- Kaon parent absent from the E-W cone (in E_off, not cone_geff) — ≤2% effect (§6 vi).
- Multi-generator (EPOS/QGSJET) E_off spread — needs regenerating the kernel
  moments with those generators (was network-blocked; retry on the cluster).
- Yield-level Barr refit for a truly unified calibrated 3D cascade — needs
  daemonflux's build code (ships splines + covariance, not the param→yield recipe).
- Affiliation placeholder in `PAPER_DRAFT.md:5` (`$^{1}$*[affiliation]*`).
- Other-site validation (needs their reference tables).

## Gotchas learned
- Always run scripts from `tools/mceq3d` with `PYTHONPATH=$PWD` (relative paths
  to `honda_kam.npz`, `.cache3d`, kernels).
- `cone_cutoff=True` is now the default → a bare `solve()` triggers a ~one-time
  finemap back-trace (memoised on the instance + cached to `cache_dir`); pass
  `cone_cutoff=False` for the fast single-cutoff approximation.
- Vault/report publishing is a macOS-side workflow (`obsidian-vault-reporter`
  skill → `~/cernbox/Research_vault`); not relevant on the cluster.

## Memory note (project fact, for continuity)
The E-W validation MUST be zenith-matched; the honest delivered result is a
~11–14% overshoot at 75°, growing to ~20–25% at the extreme horizon (87°) after
the 2026-07-05 limb fix (full-sphere cutoff map; was ~26% before). The band is
±14% at 0.5 GeV around the GSF-hybrid central. `cone_cutoff`, `r_hi=40`, and now
`finemap_rc(full_sphere=True)` are the delivered defaults. Both validation scripts
now use the 2020-01-01 epoch (shared cached map).
