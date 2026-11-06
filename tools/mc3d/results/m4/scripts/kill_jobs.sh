#!/bin/bash
pkill -f "sky3d.py.*--tag"
sleep 8
pkill -9 -f "sky3d.py.*--tag"
sleep 2
ps -u $USER -o args= | grep -c "[s]ky3d.py"
