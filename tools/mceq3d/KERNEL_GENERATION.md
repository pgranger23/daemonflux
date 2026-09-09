# High-statistics production-kernel generation — runbook

**Audience:** an autonomous agent on a compute cluster. This is self-contained;
you do not need any prior context. Follow it top to bottom.

## 0. What you are producing and why

The 3D atmospheric-cascade solver in this repo (`tools/mceq3d/`) needs the
**production angle** of secondary hadrons — the quantity 1D MCEq integrates away.
You will regenerate it by running the *same* event generators MCEq uses (through
`chromo`) in single-interaction mode and histogramming the production angle

```
theta = arcsin(p_T / p),     p = sqrt(E_sec^2 - m^2)   (the TOTAL momentum)
```

> **Angle convention (fixed 2026-09 — read this).** Until 2026-09 this pipeline
> computed `theta = arctan(p_T / p)`, i.e. it put the *total* momentum where the
> *longitudinal* one belongs. Since `arctan(u) < arcsin(u)`, that biased
> `<theta^2>` **low**. Measured on identical event samples
> (`diag_angle_convention.py`), the corrected pi+ `sqrt(<theta^2>)` is **+46% at
> `E_sec` = 0.3 GeV**, +32% at 0.5, +13% at 1, +4.6% at 2, +1.1% at 5 and +0.3%
> at 10 GeV — worst exactly in the sub-GeV region that drives the off-axis
> excess `E_off`, and far larger there than the +-12% NA61 systematic. The single
> definition now lives in `kernel_regeneration.production_angle`; every consumer
> calls it. `validate_na61.py` compares `<p_T>`, which is blind to the
> convention — use `validate_na61_angle.py` (mean *angle* vs the NA61 polar-angle
> tables) to test it. Moment files produced after the fix carry a `_v2` suffix.

There are **two deliverables**, in priority order:

1. **`m_spliced.npz` — angular *moments* (PRIMARY, drop-in).** Per-`(E_proj, x_L)`
   values of `<theta>`, `<theta^2>`, and `dN/dx_L`. This is the file the solver
   consumes today (`fokker_planck_3d.load_theta2`). **Produce this first.**
2. **`k_*_spliced.npz` — full `d2N/(dx_L dtheta)` shape (SECONDARY, future).**
   Needed only for the higher-order S_N angular operator. Higher statistics.

Everything is **embarrassingly parallel**: every `(model, secondary, energy)`
combination is an independent job.

---

## 1. Environment

```bash
python -m venv .venv && . .venv/bin/activate
pip install numpy scipy matplotlib
pip install chromo==0.10.1        # event generators (SIBYLL, UrQMD, ...)
pip install MCEq crflux           # REQUIRED for the consistency gate (step 4)
```

Verify:

```bash
python -c "import chromo, MCEq, numpy, scipy; print('ok', chromo.__version__)"
```

Get the code from the personal fork (branch **`3d-extension`** has all the 3D
code; the scripts live in `tools/mceq3d/`). Run **all commands below from inside
`tools/mceq3d/`** so the local imports resolve:

```bash
git clone -b 3d-extension https://github.com/pgranger23/daemonflux.git
cd daemonflux/tools/mceq3d
```

---

## 1b. Workflow — test small *before* you scale (do this in order)

**Do not launch the big production run first.** Work up to it in three stages,
and **submit every real run through the cluster's own job-submission service**
(SLURM `sbatch`, PBS/Torque `qsub`, LSF `bsub`, … — whatever this cluster uses;
detect it and use it). Do **not** run heavy generation on a login/head node.

1. **Smoke test (interactive, ~1 min, tiny stats).** Confirm the toolchain works
   end-to-end before requesting any allocation:
   ```bash
   python kernel_regeneration.py --backend chromo --model UrQMD34 --moments \
       --sec piplus --emin 4 --emax 80 --ne 3 --nint 500 --out smoke_low.npz
   python kernel_regeneration.py --backend chromo --model Sibyll23d --moments \
       --sec piplus --emin 80 --emax 1e6 --ne 3 --nint 500 --out smoke_high.npz
   python splice_kernels.py smoke_low.npz smoke_high.npz --transition 80 \
       --out smoke_spliced.npz
   python -c "from fokker_planck_3d import load_theta2; load_theta2('smoke_spliced.npz'); print('pipeline OK')"
   ```
   Also run the existing test suite once (`pytest -q` in `tools/mceq3d/`) — it
   should pass; this confirms the environment is sound.

2. **One small batch job.** Wrap the §4 consistency-gate command in a single job
   script and submit it through the scheduler (one core, short walltime). Verify
   the printed `shape agreement` passes (§4) **before** scaling up. This is the
   acceptance test — if it fails at small scale it will fail at large scale.

3. **Full production.** Only now submit the full grid (§3, §5), one job per
   `(model, secondary)` (and optionally per energy — see §6), as a job **array**
   through the scheduler. Size walltime/cores from §9.

A minimal SLURM example (adapt the directives and module loads to this cluster;
if it is not SLURM, translate to the local scheduler):

```bash
#!/bin/bash
#SBATCH --job-name=kern_piplus_high
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --mem=4G
#SBATCH --output=kern_%x_%j.log
cd "$SLURM_SUBMIT_DIR"
source ../../.venv/bin/activate      # the env from §1
python kernel_regeneration.py --backend chromo --model Sibyll23d --moments \
    --sec piplus --emin 80 --emax 1e6 --ne 30 --nint 200000 --out mom_piplus_high.npz
```

## 2. The fixed conventions (do not change)

- **Projectile:** proton (`2212`) — the CLI default. (Meson projectiles need a
  one-line code change; see §8. Not required for the primary deliverable.)
- **Target:** nitrogen `(14,7)` — the default. The production *angle* is
  target-independent to ~1%, so a single target suffices.
- **Secondaries (`--sec`):** `piplus`, `piminus`, `Kplus`, `Kminus`.
- **Two models, spliced** (no single model spans the energy range):
  - low energy: **`UrQMD34`** for `E_lab` ≈ 4–80 GeV;
  - high energy: **`Sibyll23d`** for `E_lab` ≳ 80 GeV
    (SIBYLL has a hard floor at √s = 10 GeV ≈ `E_lab` 53 GeV — **never run
    Sibyll23d below `--emin 80`**).
  - splice at **80 GeV**.
- **One Fortran model per Python process** (chromo uses global COMMON blocks) —
  so each model is a *separate* invocation. The CLI already respects this.

---

## 3. PRIMARY deliverable — `m_spliced.npz` (angular moments)

Statistics: `--nint 200000` per energy is far more than enough for the variance
(`<theta^2>` is a mean over ~10 secondaries/event → sub-percent at 2×10⁵). Do not
go higher for the moments file; save the big statistics for the full shape (§5).

The minimal drop-in is the dominant **proton → π⁺** channel. Run the two models,
then splice:

```bash
# low-energy moments (UrQMD34, 4-80 GeV)
python kernel_regeneration.py --backend chromo --model UrQMD34 --moments \
    --sec piplus --emin 4 --emax 80 --ne 12 --nint 200000 --out mom_piplus_low.npz

# high-energy moments (Sibyll23d, 80 GeV - 1 PeV)
python kernel_regeneration.py --backend chromo --model Sibyll23d --moments \
    --sec piplus --emin 80 --emax 1e6 --ne 30 --nint 200000 --out mom_piplus_high.npz

# splice -> the drop-in file the solver loads
python splice_kernels.py mom_piplus_low.npz mom_piplus_high.npz \
    --transition 80 --out m_spliced.npz
```

`m_spliced.npz` is now a **drop-in replacement** for the file of the same name in
the repo.

**Filenames matter — use these exact names** (they are hard-coded in the solver
and in `channel_comparison.py`; do not invent variants like `m_spliced_piplus`):

| secondary | final spliced filename            |
|-----------|-----------------------------------|
| `piplus`  | **`m_spliced.npz`** (the default drop-in loaded by ~10 modules + the tests; it *is* the π⁺ channel) |
| `piminus` | `m_piminus.npz`                   |
| `Kplus`   | `m_Kplus.npz`                     |
| `Kminus`  | `m_Kminus.npz`                    |

Repeat the §3 commands for the other three secondaries, writing each splice to the
filename above. If you prefer a self-documenting original, splice π⁺ to
`m_spliced_piplus.npz` and then `cp m_spliced_piplus.npz m_spliced.npz` — but the
file the code loads by default **must** be named `m_spliced.npz`.

### Sanity check (prints automatically)

The `--moments` run prints a `<theta>(E_sec)` table. It **must decrease
monotonically** with energy (θ ~ 1/E), landing around **~16 deg at ~1 GeV**
and falling below ~0.1° by multi-TeV. If it is flat or rises, something is wrong
(stop and report).

---

## 4. Consistency gate (mandatory — this is the acceptance test)

For the **non-`--moments`** kernel, `kernel_regeneration.py` automatically
compares the p_T- (or θ-) marginal `dN/dx_L` to **MCEq's own stored 1D yield**.
This validates the entire pipeline against MCEq before anything downstream.

Run it once per model for π⁺ (this is the gate; it needs MCEq installed):

```bash
python kernel_regeneration.py --backend chromo --model Sibyll23d \
    --sec piplus --emin 80 --emax 10000 --ne 6 --nint 100000 --out gate_high.npz
```

Read the printed lines:

```
consistency vs MCEq stored dN/dx_L:
  norm factor (MCEq convention) = <value>
  shape agreement within 5%     = <fraction>   (after removing norm factor)
  raw agreement within 5%       = <fraction>
```

**Acceptance:**
- `shape agreement within 5%` ≳ **0.7** (this is the real physics gate).
- `norm factor` is **O(1)** and roughly energy-independent. **A norm factor ≠ 1
  is expected** — it is MCEq's internal `hadr_yields` storage convention and it
  cancels in every angular quantity downstream. Just **record its value**; do not
  try to "fix" it.

If `shape agreement` is low, increase `--nint` and re-check. If the gate prints
"No reference available", **MCEq is not installed** — fix the environment (§1).

(There is no MCEq table for `UrQMD34`, so its gate returns "No reference"; that is
expected. Validate the pipeline with the SIBYLL gate above.)

---

## 5. SECONDARY deliverable — full `d2N/(dx_L dtheta)` shape

Same structure, but use `--angular` (not `--moments`) and **high statistics**
`--nint 1000000` (the forward/high-x and large-θ tail bins are rare). The angular
effect is negligible above a few TeV, so you may cap the high-energy grid at
`--emax 1e4` to save time.

```bash
python kernel_regeneration.py --backend chromo --model UrQMD34 --angular \
    --sec piplus --emin 4 --emax 80 --ne 12 --nint 1000000 \
    --ntheta 60 --theta-max-deg 40 --out k_piplus_low.npz
python kernel_regeneration.py --backend chromo --model Sibyll23d --angular \
    --sec piplus --emin 80 --emax 1e4 --ne 24 --nint 1000000 \
    --ntheta 60 --theta-max-deg 40 --out k_piplus_high.npz
python splice_kernels.py k_piplus_low.npz k_piplus_high.npz \
    --transition 80 --out k_piplus_spliced.npz
```

Repeat for the other three secondaries. The low/high θ-grids
(`--ntheta`, `--theta-max-deg`) **must match** between the two models you splice.

---

## 5b. Single-machine driver (many cores, no scheduler)

On a workstation/interactive node with tens of cores, `regen_moments_mp.py`
reproduces the whole of section 3 in one command. It uses the same generators,
energy grid (12 points 4-80 GeV UrQMD-3.4, 30 points 80 GeV-1 PeV SIBYLL-2.3d),
the same `x_L` binning and the same `--nint` per energy, but shards energies
across a local process pool and histograms **all four secondaries from one event
sample** (the section 8 optimisation):

```bash
# what was actually used for the delivered *_v2.npz set (2026-09):
python regen_moments_mp.py --nint 60000 --nint-high 200000 --nproc 46 \
    --shard-low 2000 --shard-high 20000 --suffix _v2
# -> m_spliced_v2.npz, m_piminus_v2.npz, m_Kplus_v2.npz, m_Kminus_v2.npz
```

Same-energy shards are merged by summing raw `counts` / `sum_theta` /
`sum_theta_sq` (the count-weighted merge section 6 warns about), so the result is
statistically identical to one long run. Each per-model file also carries
`theta_sq_a` / `theta_sq_b` / `counts_a` / `counts_b`: two statistically
independent halves whose difference measures the Monte-Carlo error directly.

**Cost and how much statistics you actually need.** UrQMD-3.4 dominates: ~100
evt/s/core at 4 GeV falling to ~18 evt/s/core at 80 GeV, versus ~1600-2500
evt/s/core for SIBYLL-2.3d at any energy. The 2e5-per-energy figure in section 3
is generous for the *moments*; the half-split test shows 6e4 UrQMD interactions
per energy already gives ``<theta^2>`` to **<= 2%** over `E_sec` = 0.3-10 GeV for
pi+, pi- and K+ (23 of 24 report points <= 1.4%, worst 2.0% for pi+ at 10 GeV).
The exception is **K- below ~0.5 GeV** (6-9% at 6e4, ~5% even at 2e5): sub-GeV
K- production from a <= 80 GeV proton is rare in UrQMD and no affordable
statistics fixes it. That channel is reference-only downstream (the delivered
cone uses the pion angle), so it was accepted. Wall clock for the command above
on 46 contended cores: 20 min UrQMD + 3 min SIBYLL.

Merging a top-up run into an existing per-model file (to reach the full 2e5) is
`regen_moments_mp.merge_model_files(out, fileA, fileB, ...)`, then re-splice.

Cross-check the delivered file with the moments-level normalisation gate:

```bash
python gate_moments_norm.py m_spliced_v2.npz --sec piplus   # norm ~1.0
```

## 6. Parallelization (optional, for many cores)

Use the cluster's scheduler for this — submit a **job array** (e.g. SLURM
`--array`, one task per `(model, secondary)` or per energy), not a shell loop on
the head node.

Each **energy point is independent**. To shard across cores, run single-energy
jobs (`--emin E --emax E --ne 1`) with distinct output names, then concatenate.
Combine single-energy **moment** shards into one model-level file like this:

```python
# combine_moments.py  — usage: python combine_moments.py out.npz shard1.npz shard2.npz ...
import sys, numpy as np
out, files = sys.argv[1], sys.argv[2:]
d = [dict(np.load(f)) for f in files]
order = np.argsort(np.concatenate([x["proj_energies"] for x in d]))
cat = lambda k: np.concatenate([x[k] for x in d], axis=0)[order]
np.savez(out, is_moments=True, xl_edges=d[0]["xl_edges"],
         xl_centers=d[0]["xl_centers"],
         proj_energies=np.concatenate([x["proj_energies"] for x in d])[order],
         theta_mean=cat("theta_mean"), theta_sq=cat("theta_sq"),
         dndx=cat("dndx"), e_sec=cat("e_sec"))
print("wrote", out)
```

Then splice the combined low- and high-model files with `splice_kernels.py` as in
§3. (For full-shape kernels, shard the same way and combine with `cat` over the
`kernel` array instead.)

**Do not** try to run two different chromo models in one process, and **do not**
merge same-energy shards by simple averaging of `theta_sq` unless you weight by
the per-shard counts — energy-sharding (above) avoids that entirely.

---

## 7. Output schema (what `m_spliced.npz` must contain)

A correct moments file has exactly these keys (verify before returning):

| key            | shape          | meaning                                   |
|----------------|----------------|-------------------------------------------|
| `is_moments`   | scalar `True`  | marks a moments file                      |
| `proj_energies`| `(nE,)`        | projectile lab energies [GeV]             |
| `xl_edges`     | `(n_xl+1,)`    | `x_L = E_sec/E_proj` bin edges            |
| `xl_centers`   | `(n_xl,)`      | `x_L` bin centers                         |
| `e_sec`        | `(nE, n_xl)`   | secondary energy [GeV] = `x_L * E_proj`   |
| `theta_mean`   | `(nE, n_xl)`   | `<theta>` [rad] per bin                    |
| `theta_sq`     | `(nE, n_xl)`   | `<theta^2>` [rad²] per bin (the key input)|
| `dndx`         | `(nE, n_xl)`   | `dN/dx_L` per interaction (pooling weight)|

Quick verification:

```bash
python -c "import numpy as np; d=np.load('m_spliced.npz'); \
print(sorted(d.keys())); print('nE,nxl=',d['theta_sq'].shape); \
print('E range',d['proj_energies'].min(),d['proj_energies'].max())"
```

End-to-end check that the solver accepts it (should print an E vs θ table with no
errors):

```bash
python -c "from fokker_planck_3d import load_theta2; \
e,t=load_theta2('m_spliced.npz'); \
import numpy as np; print('theta(deg) at',np.round(e[::6],2),'GeV =', \
np.round(np.degrees(np.sqrt(t[::6])),2))"
```

---

## 8. Known gotchas

- **MCEq not installed → no consistency gate.** The gate is the acceptance test;
  MCEq is mandatory (§1).
- **SIBYLL below 53 GeV silently misbehaves.** Keep `Sibyll23d --emin ≥ 80`.
- **Per-secondary regeneration cost.** The CLI re-runs events for each `--sec`.
  If runtime matters, you can histogram all four secondaries from one event
  sample by editing `ChromoSource.generate` in `kernel_regeneration.py` to return
  every species at once — optional optimization, not required.
- **Meson projectiles** (π, K re-interactions) are not exposed on the CLI. To add
  them, pass `projectile=211`/`321` to `ChromoSource(...)` (the class already
  accepts it) or add a `--proj` flag. Only needed for the complete production set,
  not the primary deliverable.
- **numpy 2.x** is fine; the scripts use `np.histogram`/`np.histogram2d` only.

---

## 9. Resource estimate (measured: SIBYLL ≈ 3000 evt/s, UrQMD ≈ 900 evt/s, 1 core)

| set | per-energy `nint` | total core-hours (4 secondaries, both models) |
|-----|-------------------|-----------------------------------------------|
| moments (`m_spliced.npz`) | 2×10⁵ | **~5** |
| full shape (`k_*_spliced.npz`) | 1×10⁶ | **~25** (one model) → ~100–200 with 5-model systematics |

SIBYLL slows at high energy (higher multiplicity); the PeV points dominate. Cap
`--emax` at `1e4` for the angular shape (no angular signal above) to cut this.

---

## 10. What to return

1. `m_spliced.npz` (π⁺) and, if produced, `m_spliced_{piminus,Kplus,Kminus}.npz`.
2. The full-shape `k_*_spliced.npz` files (if step §5 was run).
3. The **printed consistency-gate numbers** from §4 (norm factor + shape
   agreement), and the `<theta>(E_sec)` sanity table from §3.
4. A note of: chromo version, models/energies/`nint` used, total wall-clock.

Optional: add `--plot` to any non-moments run to also emit a `.png` of the kernel
and the marginal-vs-MCEq overlay — useful evidence to return.
```
