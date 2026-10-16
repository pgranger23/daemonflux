#!/bin/bash
until [ -f /home/pigrange/mc3d_work/runs/m3b/sky/pilot_total.npz ]; do sleep 30; done
/home/pigrange/mc3d_work/runs/m3b/scripts/run_decomp.sh
