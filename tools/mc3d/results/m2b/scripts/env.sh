# source me: environment for the mc3d milestone-2b runs
export PY=/cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc14-opt/bin/python3.12
export R=/afs/cern.ch/work/p/pigrange/daemonflux
export W=/home/pigrange/mc3d_work
export M=$W/runs/m2b
export PYTHONPATH=$W/pylib:$R/.venv/lib/python3.12/site-packages:$R/src:$R/tools/mceq3d:$R/tools/mc3d
export MC3D_POOL_CACHE=$W/pools
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
