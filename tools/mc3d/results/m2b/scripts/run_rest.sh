#!/bin/bash
# milestone-2b: patch gate g40/g55 and the collinear twin p1d (sec. 7, adapted)
source /home/pigrange/mc3d_work/runs/m2b/scripts/env.sh
cd $M
SK=$R/tools/mc3d/sky.py
C="--nproc 30 --cache $W/pools --chunk 6000 --nstrata 6 --outdir $M/sky"
$PY $SK --nshower 250000  $C --tag g40 --seed 777 --theta-inj 40 > $M/logs/g40.log 2>&1
$PY $SK --nshower 250000  $C --tag g55 --seed 777 --theta-inj 55 > $M/logs/g55.log 2>&1
$PY $SK --nshower 1500000 $C --tag p1d --seed 20260910 --collinear > $M/logs/p1d.log 2>&1
echo done > $M/logs/rest.done
