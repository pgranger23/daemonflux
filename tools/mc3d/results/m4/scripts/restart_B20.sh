#!/bin/bash
source /home/pigrange/mc3d_work/runs/m4/env.sh
cd $M
L=$(grep "running:" logs/progress_B.txt | tail -n 1 | sed 's/.*finished tasks) \([0-9.]*\) core-h.*/\1/')
pkill -f "m4_driver.py B"; sleep 1
pkill -f "sky3d.py.*--tag B"; sleep 8; pkill -9 -f "sky3d.py.*--tag B"; sleep 2
echo $L > work/cpu_prior_B.txt
echo 0 > work/workers_addon.txt
nohup scripts/launch_B.sh > logs/driver_B.out 2>&1 &
sleep 5; echo prior $L
