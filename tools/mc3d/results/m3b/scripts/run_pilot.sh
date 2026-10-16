#!/bin/bash
source /home/pigrange/mc3d_work/runs/m3b/env.sh
cd $M/work
$PY $R/tools/mc3d/sky3d.py --nshower 0 --alloc $M/reports/alloc_pilot.npz --nproc 34 \
    --cache $MC3D_POOL_CACHE --chunk 1600 --outdir $M/sky --work $M/work \
    --seed 20261012 --tag pilot --r-thr 65 > $M/logs/pilot.log 2>&1
