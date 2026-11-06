#!/bin/bash
source /home/pigrange/mc3d_work/runs/m4/env.sh
cd $M/work
D=$R/tools/mc3d/results/m4/scripts
$PY - <<'PY'
import numpy as np, os
M=os.environ["M"]; a=dict(np.load(M+"/reports/alloc_B.npz")); a["n"]=np.maximum(800,(a["n"]*0.5).astype(int)); np.savez(M+"/reports/alloc_addon.npz",**a); print("addon primaries",a["n"].sum())
PY
AL="--nshower 0 --alloc $M/reports/alloc_addon.npz --nproc NPROC --cache $MC3D_POOL_CACHE --chunk 1600 --outdir $M/sky --work $M/work --seed 27182818 --tag addon --model GSFG:GH --r-thr 0 --no-cutoff --no-bend-primary --mu-cf --no-keep-cfiles"
exec $PY $D/m4_driver.py addon 50 $M/work/workers_B.txt -- $PY $R/tools/mc3d/sky3d.py $AL
