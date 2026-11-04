#!/bin/bash
source /home/pigrange/mc3d_work/runs/m3d/env.sh
cd $M/work
S=$M/scripts/closure_floor.py
GUARD_MASS=0.94 $PY $S chromo 0 2000 8 $M/reports/cl_Bold_100.npz 100 >> $M/logs/closure_B.log 2>&1
$PY $S chromo 0 2000 8 $M/reports/cl_B_100.npz 100 >> $M/logs/closure_B.log 2>&1
GUARD_MASS=0.94 $PY $S chromo 0 1500 8 $M/reports/cl_Bold_300.npz 300 >> $M/logs/closure_B.log 2>&1
$PY $S chromo 0 1500 8 $M/reports/cl_B_300.npz 300 >> $M/logs/closure_B.log 2>&1
echo "$(date +%T) done gfix 100/300" >> $M/logs/closure.log
