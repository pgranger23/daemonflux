#!/bin/bash
source /home/pigrange/mc3d_work/runs/m2b/scripts/env.sh
until [ -f $M/logs/rest.done ]; do sleep 5; done
exec $M/scripts/run_x10.sh
