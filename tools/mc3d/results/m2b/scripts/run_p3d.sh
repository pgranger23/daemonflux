#!/bin/bash
# milestone-2b pilot, PHASE2 sec. 7 command with paths adapted (30 workers)
source /home/pigrange/mc3d_work/runs/m2b/scripts/env.sh
cd $M
SK=$R/tools/mc3d/sky.py
C="--nproc 30 --cache $W/pools --chunk 6000 --nstrata 6 --outdir $M/sky"
$PY $SK --nshower 1500000 $C --tag p3d --seed 20260910
