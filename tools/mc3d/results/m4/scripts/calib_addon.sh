#!/bin/bash
source /home/pigrange/mc3d_work/runs/m4/env.sh
cd $M/work
/usr/bin/time -f "%U %S addon_cal" -a -o $M/logs/cpu_cal.txt $PY $R/tools/mc3d/sky3d.py --nshower 0 --alloc $M/reports/alloc_cal_addon.npz --nproc 6 --cache $MC3D_POOL_CACHE --chunk 1600 --outdir $M/sky --work $M/work --seed 27182818 --tag addon_cal --model GSFG:GH --r-thr 0 --no-cutoff --no-bend-primary --mu-cf --no-keep-cfiles > $M/logs/addon_cal.log 2>&1
