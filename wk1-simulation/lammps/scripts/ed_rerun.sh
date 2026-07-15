#!/bin/bash
# E_d sweeps v3: ramp scan (non-monotonicity-safe), 20 directions
set -e
cd "$(dirname "$0")/../.."
LMP=$HOME/miniforge3/envs/lammps/bin/lmp
PY=$HOME/miniforge3/envs/lammps/bin/python3

$PY ed.py Fe --np 4 --lmp "$LMP"
$PY ed.py W --np 4 --lmp "$LMP"
echo ED_FEW_DONE
