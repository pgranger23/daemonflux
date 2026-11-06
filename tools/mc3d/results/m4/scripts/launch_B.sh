#!/bin/bash
source /home/pigrange/mc3d_work/runs/m4/env.sh
cd $M/work
D=$R/tools/mc3d/results/m4/scripts
[ -f $M/work/workers_addon.txt ] || echo 0 > $M/work/workers_addon.txt
BL="--nshower 0 --alloc $M/reports/alloc_B.npz --nproc NPROC --cache $MC3D_POOL_CACHE --chunk 1600 --outdir $M/sky --work $M/work --seed 31415926 --tag B --model GSFG:GH --r-thr 65 --resim-rmin 32 --resim-groups p --reanchor"
exec $PY $D/m4_driver.py B 285 $M/work/workers_addon.txt -- $PY $R/tools/mc3d/sky3d.py $BL
