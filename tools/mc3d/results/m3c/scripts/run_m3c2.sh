#!/bin/bash
# milestone 3c, second queue (2026-11-04 12:07): replaces the tail of
# run_m3c.sh (a1vert / a1horiz dropped in favour of the no-primary-bend
# trio, which tests H1 on an exactly flat azimuthal pattern and the decay
# pipeline).  Waits for the c3anc job started by run_m3c.sh (pid $1).
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
run c3res      $T --r-thr 65
run a1ncnb     --nshower 1000000 $C --r-thr 0 --no-cutoff --no-bend-primary
run a1ncnb0    --nshower 1000000 $C --r-thr 0 --no-cutoff --no-bend-primary --mu-bend none
run a1ncnbnomu --nshower 1000000 $C --r-thr 0 --no-cutoff --no-bend-primary --no-muon-field
echo "$(date '+%F %T') queue 2 all done" >> $P
