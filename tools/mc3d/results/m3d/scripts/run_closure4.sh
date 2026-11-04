#!/bin/bash
source /home/pigrange/mc3d_work/runs/m3d/env.sh
cd $M/work
S=$M/scripts/closure_floor.py
GUARD_MASS=0.94 $PY $S chromo 0 4000 8 $M/reports/cl_Bold_40.npz 40 >> $M/logs/closure_B.log 2>&1
$PY $S chromo 0 4000 8 $M/reports/cl_B_40.npz 40 >> $M/logs/closure_B.log 2>&1
# finer energy grid, both guards, 12000 showers
for E in "1.2 1.8 2.5" "3.5 5 7"; do
GUARD_MASS=0.94 $PY $S chromo 0 12000 8 $M/reports/cl_Bold_g$(echo $E|tr ' ' _).npz $E >> $M/logs/closure_B.log 2>&1
$PY $S chromo 0 12000 8 $M/reports/cl_B_g$(echo $E|tr ' ' _).npz $E >> $M/logs/closure_B.log 2>&1
done
echo "$(date +%T) done gfix inputs" >> $M/logs/closure.log
