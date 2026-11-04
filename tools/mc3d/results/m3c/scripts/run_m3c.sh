#!/bin/bash
# milestone 3c runs, launched 2026-11-04 with nohup (sequential, 24 cores).
# Task 2 (muon-bend mechanism): paired with m3b's a1anc / a1nomu (same seeds,
# tasks, allocation: 1e6 primaries, E^-0.8, r_thr 0, re-anchored).
# Task 3 (re-anchor vs re-simulate): common random numbers (--crn), the pilot
# allocation x 0.2, the same seed for both modes.
source /home/pigrange/mc3d_work/runs/m3c/env.sh
cd $M/work
SK=$R/tools/mc3d/sky3d.py
P="$M/logs/progress.txt"
C="--nproc 24 --cache $MC3D_POOL_CACHE --chunk 10000 --nstrata 7 --outdir $M/sky --work $M/work --seed 31001"
T="--nshower 0 --alloc $M/reports/alloc_m3c_0p2.npz --nproc 24 --cache $MC3D_POOL_CACHE --chunk 1600 --outdir $M/sky --work $M/work --seed 20261104 --crn --reanchor"
run() {   # tag, args...
  local tag=$1; shift
  echo "$(date '+%F %T') start $tag: $*" >> $P
  if $PY $SK "$@" --tag $tag > $M/logs/$tag.log 2>&1; then
    echo "$(date '+%F %T') done  $tag" >> $P
  else
    echo "$(date '+%F %T') FAILED $tag (exit $?)" >> $P
  fi
}
run a1nc     --nshower 1000000 $C --r-thr 0 --reanchor --no-cutoff
run a1ncnomu --nshower 1000000 $C --r-thr 0 --reanchor --no-cutoff --no-muon-field
run c3anc    $T --r-thr 0
run c3res    $T --r-thr 65
run a1vert   --nshower 1000000 $C --r-thr 0 --reanchor --mu-bend vertical
run a1horiz  --nshower 1000000 $C --r-thr 0 --reanchor --mu-bend horizontal
echo "$(date '+%F %T') all done" >> $P
