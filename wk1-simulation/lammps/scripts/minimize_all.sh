#!/bin/bash
# 0 K minimization of all equil/cascade restart pairs -> per-atom PE dumps
set -e
cd "$(dirname "$0")/.."
LMP=$HOME/miniforge3/envs/lammps/bin/lmp

run() {  # material snap tag
    mpirun -np 6 "$LMP" -in "inputs/in.minimize_$1" \
        -log "outputs/logs/log.min_$1_$3" -var snap "outputs/$2" -var tag "$3"
}

run SiC sic_equil.restart equil
run SiC cascade_SiC_1.restart cascade1
run Fe fe_equil.restart equil
run Fe cascade_Fe_1.restart cascade1
run W w_equil.restart equil
run W cascade_W_1.restart cascade1
echo ALL_MINIMIZATIONS_DONE
