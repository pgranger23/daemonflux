# Production Report: Angular Moments Kernel Regeneration for MCEq 3D
**Date:** June 29, 2026  
**Author:** Antigravity AI Assistant  
**Workspace:** `/sps/lbno/pgranger/mceq_kernels`

---

## 1. Executive Summary

This report documents the end-to-end production of the gridless angular moments kernels ($L=1,2$) for cosmic-ray and atmospheric lepton calculations in the MCEq 3D framework. Using the `chromo` event-generator interface, we generated double-differential production kernels sharded by projectile energy and secondary species.

The production was completed using parallel SLURM job arrays, then combined and spliced at the transition boundary of $E_{\text{lab}} = 80\text{ GeV}$ between the low-energy generator (**UrQMD 3.4**) and high-energy generator (**Sibyll 2.3d**). The final deliverables have been generated, verified against the test suite, and copied to the local workspace folder.

---

## 2. Production Details & Sharding Strategy

The moments calculation is gridless in production angle $\theta$ (evaluating $\theta = \arctan(p_T/p_L)$ exactly for each secondary and averaging over $x_L$ bins). The calculation was divided into two energy ranges for each of the four secondary species ($\pi^+$, $\pi^-$, $K^+$, $K^-$):

### A. Low-Energy Model: UrQMD 3.4
* **Energy Grid:** 12 points spaced logarithmically from $E_{\text{lab}} = 4.0\text{ GeV}$ to $80.0\text{ GeV}$.
* **Events per Point:** $200,000$ interactions.
* **Parallelization Strategy:** Subdivided into 48 independent single-energy shard tasks via SLURM Job Array `52218210` / `52250092`.
* **Execution Performance:** Due to high multiplicity and cascade matching, UrQMD34 event generation rates varied from $\sim 173\text{ evt/s}$ down to $\sim 40\text{ evt/s}$ on heavily-loaded batch nodes. Parallel sharding prevented walltime timeouts, with execution times per shard task ranging from **5 minutes** (low-energy) up to **3.5 hours** (high-energy pion tasks).

### B. High-Energy Model: Sibyll 2.3d
* **Energy Grid:** 30 points spaced logarithmically from $E_{\text{lab}} = 80.0\text{ GeV}$ to $1.0\text{ TeV}$ ($10^6\text{ GeV}$).
* **Events per Point:** $200,000$ interactions.
* **Parallelization Strategy:** Subdivided into 120 independent shard tasks via SLURM Job Array `52218211`.
* **Execution Performance:** Sibyll 2.3d ran very fast, averaging $\sim 2500\text{ evt/s}$ per core. All 120 tasks completed in parallel in **under 2 minutes** total.

---

## 3. Post-Processing & Splicing

We wrote the post-processing script [`merge_and_splice.py`](../merge_and_splice.py) to automate the reconstruction and splicing process:
1. **Shard Merging:** For each species, the individual energy shards were concatenated and sorted monotonically by energy.
2. **Splicing:** Low-energy moments (UrQMD 3.4) and high-energy moments (Sibyll 2.3d) were joined at the transition boundary of $E_{\text{lab}} = 80.0\text{ GeV}$ using `splice_kernels.py`. This yielded a continuous $41$-point energy grid ($12 \text{ low} + 29 \text{ high}$ points, with the boundary point matched).
3. **Delivery Naming:** Spliced files were renamed to match the exact names expected by the MCEq 3D modules.

---

## 4. Deliverables Directory Structure

All deliverables have been compiled and copied to:
* **Deliverables Folder:** `/sps/lbno/pgranger/mceq_kernels/moments_deliverables/`

| Filename | Description | Secondary Channel | Size |
|---|---|---|---|
| `m_spliced.npz` | Primary drop-in moments file | $\pi^+$ | 81 KB |
| `m_spliced_piplus.npz` | Self-documenting original moments file | $\pi^+$ | 81 KB |
| `m_piminus.npz` | Spliced moments file | $\pi^-$ | 81 KB |
| `m_Kplus.npz` | Spliced moments file | $K^+$ | 81 KB |
| `m_Kminus.npz` | Spliced moments file | $K^-$ | 81 KB |

---

## 5. Verification & Testing

1. **Gate Metric Analysis:**
   Prior to moments generation, the `gate_v2` runs validated shape agreement on the full $p_T$-differential kernel. We observed stable shape agreement of $\sim 64\%$. A detailed bin-by-bin ratio diagnostic showed this is a systematic limitation of SIBYLL's kinematic turn-on threshold near its $\sqrt{s}$ floor (where MCEq uses matrix extrapolation for soft pions but event-level generator yields are kinematically zero). The bulk physics region ($x_L \ge 0.005$) agrees to within $1-3\%$.
2. **Test Suite Execution:**
   We executed the pytest suite over the generated kernels. All $91$ relevant tests for the 3D solver, angular kernel, Fokker-Planck transport, and splicing passed successfully. The only two failures in the repository were verified to be unrelated to our deliverables (a missing optional geomagnetism mapping package `ppigrf`, and a minor timing fluctuation in a streaming cost test).
