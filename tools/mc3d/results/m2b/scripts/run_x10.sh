#!/bin/bash
# milestone-2b statistics run: 10x the pilot (1.5e7 showers), new seeds,
# chunk x10 so the task count stays ~1700; p3dx then its collinear twin p1dx
source /home/pigrange/mc3d_work/runs/m2b/scripts/env.sh
cd $M
SK=$R/tools/mc3d/sky.py
C="--nproc 30 --cache $W/pools --chunk 60000 --nstrata 6 --outdir $M/sky"
$PY $SK --nshower 15000000 $C --tag p3dx --seed 20261008 > $M/logs/p3dx.log 2>&1
$PY $SK --nshower 15000000 $C --tag p1dx --seed 20261008 --collinear > $M/logs/p1dx.log 2>&1
echo done > $M/logs/x10.done
