Testing nuclear materials via simulation, Garcia SRP 2026
## What it is
10 keV PKA displacement cascades via LAMMPS simulation and OVITO visualization: 3C-SiC, Fe, and W
## Setup
Needs an MPI build of LAMMPS, and Python 3.10-3.12
```py
pip install -r analysis/requirements.txt
export QT_QPA_PLATFORM=offscreen   # if headless
```
## Run
```py
python3 run.py Fe --runs 1 --np 4
```
## Figure generation
```py
python3 analysis/thermo.py lammps/outputs/log.run_Fe_cascade_1 --outdir figures/thermo
python3 analysis/evolution.py lammps/outputs/traj_Fe_spike_1.lammpstrj
    --reference lammps/structures/perfect_lattice_Fe.data --out figures/defects/Fe_run1.png
python3 analysis/render.py lammps/outputs/traj_Fe_spike_1.lammpstrj
    --reference lammps/structures/perfect_lattice_Fe.data --material Fe --energy "10 keV"
```
