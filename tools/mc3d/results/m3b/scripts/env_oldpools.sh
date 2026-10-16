# the milestone-2b environment (buggy SIBYLL nucleon pools): regression gate C1 and runs a0rot/a0anc only
export PY=/cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc14-opt/bin/python3.12
export R=/afs/cern.ch/work/p/pigrange/daemonflux
export W=/home/pigrange/mc3d_work
export PYTHONPATH=$W/pylib:$R/.venv/lib/python3.12/site-packages:$R/src:$R/tools/mceq3d:$R/tools/mc3d
export MC3D_POOL_CACHE=$W/pools OMP_NUM_THREADS=1
export M=$W/runs/m3b
export MC3D_FIELD_CACHE=$M/work/fieldcache
