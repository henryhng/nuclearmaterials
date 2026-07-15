#!/bin/bash
# Phase 2+3, strictly sequential: SiC relax extension then E_d sweeps
set -e
cd "$(dirname "$0")/.."
LMP=$HOME/miniforge3/envs/lammps/bin/lmp
PY=$HOME/miniforge3/envs/lammps/bin/python3

mpirun -np 6 "$LMP" -in inputs/in.extend_SiC_relax \
    -log outputs/logs/log.run_SiC_relaxext_1 -var runid 1
echo EXTENSION_DONE

cd ..
$PY ed.py Fe --np 4 --lmp "$LMP"
$PY ed.py W --np 4 --lmp "$LMP"
$PY ed.py SiC --np 4 --lmp "$LMP"
echo ED_ALL_DONE
