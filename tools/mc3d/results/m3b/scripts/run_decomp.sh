#!/bin/bash
# mechanism decomposition, paired with a1anc / a1rot (same seeds and tasks)
source /home/pigrange/mc3d_work/runs/m3b/env.sh
cd $M/work
SK=$R/tools/mc3d/sky3d.py
C="--nproc 34 --cache $MC3D_POOL_CACHE --chunk 10000 --nstrata 7 --outdir $M/sky --work $M/work --seed 31001"
$PY $SK --nshower 1000000 $C --tag a1nomu --r-thr 0 --reanchor --no-muon-field > $M/logs/a1nomu.log 2>&1
$PY $SK --nshower 1000000 $C --tag a1nobend --r-thr 0 --no-bend-primary > $M/logs/a1nobend.log 2>&1
