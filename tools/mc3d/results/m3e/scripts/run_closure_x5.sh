#!/bin/bash
source /home/pigrange/mc3d_work/runs/m3e/env.sh
cd $M/work
S=$M/scripts/closure_floor.py
t() { /usr/bin/time -f "%U %S $1" -a -o $M/logs/cpu_closure.txt "${@:2}"; }
for cfg in "mceq 0 A1" "kinematic 0 A2" "chromo 0 B"; do
  set -- $cfg
  t $3lo_x5 $PY $S $1 $2 40000 16 $M/reports/cl_$3_lo_x5.npz 1.5 2.5 4 >> $M/logs/closure_$3.log 2>&1
  t $3hi_x5 $PY $S $1 $2 20000 16 $M/reports/cl_$3_hi_x5.npz 6 10 20 >> $M/logs/closure_$3.log 2>&1
done
t B100_x4 $PY $S chromo 0 6000 16 $M/reports/cl_B_100_x4.npz 100 >> $M/logs/closure_B.log 2>&1
t A2_100_x4 $PY $S kinematic 0 6000 16 $M/reports/cl_A2_100_x4.npz 100 >> $M/logs/closure_A2.log 2>&1
echo done > $M/logs/closure_x5.done
