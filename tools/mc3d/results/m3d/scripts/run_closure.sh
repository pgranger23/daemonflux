#!/bin/bash
source /home/pigrange/mc3d_work/runs/m3d/env.sh
cd $M/work
S=$M/scripts/closure_floor.py
for cfg in "mceq 0 A1" "kinematic 0 A2" "kinematic 2.0 A2f2" "chromo 0 B"; do
  set -- $cfg
  echo "$(date +%T) start $3" >> $M/logs/closure.log
  $PY $S $1 $2 8000 8 $M/reports/cl_$3_lo.npz 1.5 2.5 4 >> $M/logs/closure_$3.log 2>&1
  $PY $S $1 $2 4000 8 $M/reports/cl_$3_hi.npz 6 10 20 >> $M/logs/closure_$3.log 2>&1
  echo "$(date +%T) done $3" >> $M/logs/closure.log
done
