#!/bin/zsh
# Cap solves of stage 1 at M = 8 (GATES.md, entry of 2026-11-24, section 4, item 3):
#   cap10_B  cap of 10 deg on grid B (5 x 1.25 deg)   G2.3, G2.1, G2.4(a)
#   cap5_B   cap of 5 deg on grid B                    G2.3(g)
#   cap10_A  cap of 10 deg on grid A (5 x 2.5 deg)     G2.4(a)
# Each boundary: 17 tangents (orders 0..8, cos and sin parts) in four processes of
# 5, 4, 4, 4 tangents, one process at a time on cores 0-3, each under /usr/bin/time -v.
# Projected (cost runs and site solve of 2026-11-23): 61.8 core-min, 20 GB, about
# 24 min wall for 5 tangents; 52.1 core-min, 17.2 GB, about 19 min for 4; one boundary
# about 3.6 core-h and 1 h 20 wall; the three about 10.9 core-h and 4 h 10 wall.
# A process does not start with less than 28 GB available (stop, exit 3); outputs
# are never overwritten (run_solve3d.py refuses). After the four processes of a
# boundary its cells are combined at M = 8.
#
# Usage: zsh tools/cascade3d/gates/run_caps_M8.sh [cap10_B cap5_B cap10_A]
source /home/pigrange/mc3d_work/runs/native3d/env.sh
OUT=$N3/c2/solver/caps_M8
G=$R/tools/cascade3d/gates
typeset -A CAP SP
CAP=(cap10_B 10 cap5_B 5 cap10_A 10)
SP=(cap10_B 5,1.25 cap5_B 5,1.25 cap10_A 5,2.5)
BATCH=("0c,1c,1s,2c,2s" "3c,3s,4c,4s" "5c,5s,6c,6s" "7c,7s,8c,8s")
which=("$@")
(( ${#which} )) || which=(cap10_B cap5_B cap10_A)
for b in $which; do
  D=$OUT/$b
  mkdir -p $D
  (cd $R && git rev-parse HEAD && git status --short tools/cascade3d) > $D/git_head.txt
  for i in 1 2 3 4; do
    [ -e $D/tangents_b$i.npz ] && { echo "$b b$i exists, skipped"; continue; }
    avail=$(free -g | awk '/Mem:/{print $7}')
    echo "$b b$i start $(date +%T) available ${avail} GB" \
      "load $(cut -d' ' -f1-3 /proc/loadavg)"
    if [ $avail -lt 28 ]; then
      echo "$b b$i: less than 28 GB available, stop"; exit 3
    fi
    taskset -c 0-3 /usr/bin/time -v -o $D/b$i.time $PY $G/run_solve3d.py run $D \
      --cap ${CAP[$b]} --spacing ${SP[$b]} --M 8 --orders ${BATCH[$i]} \
      --batch 5 --tag b$i > $D/b$i.log 2>&1
    rc=$?
    echo "$b b$i exit $rc $(date +%T)"
    grep -E "User time|System time|Maximum resident|Elapsed" $D/b$i.time
    [ $rc -eq 0 ] || exit $rc
  done
  [ -e $D/cells_M8.npz ] || taskset -c 0-3 $PY $G/run_solve3d.py combine $D \
    --tags b1,b2,b3,b4 --M 8 --partial "" --out cells_M8.npz
done
