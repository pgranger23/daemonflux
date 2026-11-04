#!/bin/bash
source /home/pigrange/mc3d_work/runs/m3d/env.sh
cd $M/work
S=$M/scripts/closure_floor.py
echo "$(date +%T) start B (fixed guard)" >> $M/logs/closure.log
$PY $S chromo 0 8000 8 $M/reports/cl_B_lo.npz 1.5 2.5 4 >> $M/logs/closure_B.log 2>&1
$PY $S chromo 0 4000 8 $M/reports/cl_B_hi.npz 6 10 20 >> $M/logs/closure_B.log 2>&1
echo "$(date +%T) done B (fixed guard)" >> $M/logs/closure.log
$PY $M/scripts/first_gen.py > $M/logs/first_gen.log 2>&1
