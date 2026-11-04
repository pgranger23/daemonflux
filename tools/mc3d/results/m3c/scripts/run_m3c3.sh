#!/bin/bash
# milestone 3c, third queue (2026-11-04 12:12): queue 2 stopped before c3res to add
# --mu-cf (zero-field counterfactual).  a1vert/a1horiz and a1ncnb0 dropped: the
# counterfactual gives the bend vs fate decomposition in-run on the flat
# pattern (a1ncnb) and with the cutoff (a1ancd, paired with a1anc).  Waits for pid $1.
source /home/pigrange/mc3d_work/runs/m3c/env.sh
cd $M/work
SK=$R/tools/mc3d/sky3d.py
P="$M/logs/progress.txt"
C="--nproc 24 --cache $MC3D_POOL_CACHE --chunk 10000 --nstrata 7 --outdir $M/sky --work $M/work --seed 31001"
T="--nshower 0 --alloc $M/reports/alloc_m3c_0p2.npz --nproc 24 --cache $MC3D_POOL_CACHE --chunk 1600 --outdir $M/sky --work $M/work --seed 20261104 --crn --reanchor"
while kill -0 $1 2>/dev/null; do sleep 20; done
if [ -f $M/sky/c3anc_total.npz ]; then echo "$(date '+%F %T') done  c3anc (pid $1)" >> $P
else echo "$(date '+%F %T') c3anc pid $1 ended without a total -- will resume" >> $P; fi
run() {   # tag, args...
  local tag=$1; shift
  echo "$(date '+%F %T') start $tag: $*" >> $P
  if $PY $SK "$@" --tag $tag >> $M/logs/$tag.log 2>&1; then
    echo "$(date '+%F %T') done  $tag" >> $P
  else
    echo "$(date '+%F %T') FAILED $tag (exit $?)" >> $P
  fi
}
[ -f $M/sky/c3anc_total.npz ] || run c3anc $T --r-thr 0
run a1ncnb     --nshower 1000000 $C --r-thr 0 --no-cutoff --no-bend-primary --mu-cf
run a1ancd     --nshower 1000000 $C --r-thr 0 --reanchor --mu-cf
run c3res      $T --r-thr 65
run a1ncnbnomu --nshower 1000000 $C --r-thr 0 --no-cutoff --no-bend-primary --no-muon-field
echo "$(date '+%F %T') queue 3 all done" >> $P
