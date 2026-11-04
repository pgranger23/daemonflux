#!/bin/bash
source /home/pigrange/mc3d_work/runs/m3e/env.sh
cd $M/work
SK=$R/tools/mc3d/sky3d.py
H="--cache $MC3D_POOL_CACHE --chunk 1600 --outdir $M/sky --work $M/work --nproc 21 --model GSFG:GH --r-thr 65 --resim-rmin 32 --resim-groups p --reanchor"
echo "$(date +%T) start comp" >> $M/logs/progress.txt
/usr/bin/time -f "%U %S comp" -a -o $M/logs/cpu_prod.txt $PY $SK --nshower 0 --alloc $M/reports/alloc_comp.npz --seed 20270201 --tag comp $H --no-muon-field >> $M/logs/comp.log 2>&1
echo "$(date +%T) done comp" >> $M/logs/progress.txt
