#!/bin/bash
source /home/pigrange/mc3d_work/runs/m3b/env.sh
cd $M/work
SK=$R/tools/mc3d/sky3d.py
C="--nproc 34 --cache $MC3D_POOL_CACHE --chunk 10000 --nstrata 7 --outdir $M/sky --work $M/work --seed 31001"
$PY $SK --nshower 1000000 $C --tag a1rot --r-thr 0 > $M/logs/a1rot.log 2>&1
$PY $SK --nshower 1000000 $C --tag a1anc --r-thr 0 --reanchor > $M/logs/a1anc.log 2>&1
