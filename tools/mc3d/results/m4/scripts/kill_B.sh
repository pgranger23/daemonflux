#!/bin/bash
pkill -f "m4_driver.py B"
pkill -f "sky3d.py.*--tag B"
sleep 6
pkill -9 -f "sky3d.py.*--tag B"
