#!/bin/bash
source /home/pigrange/mc3d_work/runs/m3d/env.sh
cd $M/work
S=$M/scripts/closure_floor.py
for cfg in "mceq 0 A1" "kinematic 2.0 A2f2" "chromo 0 B"; do
  set -- $cfg
  echo "$(date +%T) start $3 fine" >> $M/logs/closure.log
  $PY $S $1 $2 20000 8 $M/reports/cl_$3_fine.npz 1.2 1.8 2.2 3.0 3.5 >> $M/logs/closure_$3.log 2>&1
done
# floor test, same seeds as the no-floor rung (deterministic): high statistics
for cfg in "kinematic 0 A2" "kinematic 2.0 A2f2"; do
  set -- $cfg
  $PY $S $1 $2 24000 8 $M/reports/cl_$3_floor.npz 4 6 10 >> $M/logs/closure_$3.log 2>&1
done
echo "$(date +%T) done fine + floor" >> $M/logs/closure.log
