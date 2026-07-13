#!/usr/bin/env python3
"""Run the full cascade pipeline: equilibrate, ensemble, defect counts.

    python3 run.py Fe --runs 10 --np 8
"""

import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm

HERE = Path(__file__).resolve().parent

MATERIALS = {
    "SiC": {"restart": "sic_equil.restart",
            "reference": "perfect_lattice_SiC.data",
            "traj": "traj_SiC_relax_{k}.lammpstrj",
            "pkaid": 196737,  # Si atom, box center (deck velocity is Si)
            "equil_steps": 5000, "cascade_steps": 74000},
    "Fe": {"restart": "fe_equil.restart",
           "reference": "perfect_lattice_Fe.data",
           "traj": "traj_Fe_relax_{k}.lammpstrj",
           "equil_steps": 5000, "cascade_steps": 30000},
    "W": {"restart": "w_equil.restart",
          "reference": "perfect_lattice_W.data",
          "traj": "traj_W_relax_{k}.lammpstrj",
          "equil_steps": 5000, "cascade_steps": 30000},
}


def last_step(log_path: Path) -> int:
    try:
        lines = log_path.read_text().splitlines()
    except OSError:
        return 0
    step = 0
    for line in lines:
        parts = line.split()
        if not (parts and parts[0].isdigit() and len(parts) >= 5):
            continue
        try:                          # thermo rows are all-numeric
            [float(p) for p in parts]
        except ValueError:
            continue
        step = max(step, int(parts[0]))
    return step


def run_deck(deck: str, label: str, nsteps: int, args, cwd: Path,
             lmp_vars=None) -> None:
    logname = label.split("/")[0].strip().replace(" ", "_")
    log = cwd / "outputs" / "logs" / f"log.run_{args.material}_{logname}"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["mpirun", "-np", str(args.np), args.lmp,
           "-in", deck, "-log", str(log)]
    for name, val in (lmp_vars or {}).items():
        cmd += ["-var", name, str(val)]

    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT)
    base = None
    with tqdm(total=nsteps, desc=label, unit="step",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                         "[{elapsed}<{remaining}]{postfix}") as bar:
        bar.set_postfix_str("setup/minimize")
        while proc.poll() is None:
            time.sleep(3)
            step = last_step(log)
            if step:
                base = step if base is None else base
                if step - base > 0:
                    bar.set_postfix_str("")
                bar.n = min(max(step - base, 0), nsteps)
                bar.refresh()
        bar.set_postfix_str("")
        bar.n = nsteps
        bar.refresh()
    if proc.returncode != 0:
        sys.exit(f"{label} failed, see {log}")


CHANNEL_AXES = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
CHANNEL_AXES = [[c / sum(a * a for a in ax) ** 0.5 for c in ax]
                for ax in CHANNEL_AXES]


def pka_direction(rng: random.Random):
    """Random unit vector >15 deg from all <100>/<110>/<111> channels."""
    while True:
        v = [rng.gauss(0, 1) for _ in range(3)]
        norm = sum(x * x for x in v) ** 0.5
        v = [abs(x) / norm for x in v]
        if all(sum(a * b for a, b in zip(v, ax)) < 0.966
               for ax in CHANNEL_AXES):
            return v


def analysis_python() -> str:
    """OVITO lives in the sprouster env; LAMMPS in this one. Bridge to it."""
    cand = Path.home() / "miniforge3/envs/sprouster/bin/python"
    return str(cand) if cand.exists() else sys.executable


def natoms_from_data(path: Path) -> int:
    for line in path.read_text().splitlines():
        if line.strip().endswith("atoms"):
            return int(line.split()[0])
    raise ValueError(f"no atom count in {path}")


def main():
    parser = argparse.ArgumentParser(description="Cascade ensemble runner.")
    parser.add_argument("material", choices=sorted(MATERIALS))
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--np", type=int, default=8, help="MPI ranks")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--lmp", default="lmp_mpi")
    parser.add_argument("--workdir", type=Path, default=HERE / "lammps")
    parser.add_argument("--analyze-only", action="store_true",
                        help="skip simulation, just count defects")
    args = parser.parse_args()

    mat = MATERIALS[args.material]
    cwd = args.workdir
    outputs = cwd / "outputs"
    structures = cwd / "structures"
    outputs.mkdir(exist_ok=True)
    (outputs / "results").mkdir(exist_ok=True)
    structures.mkdir(exist_ok=True)

    if not args.analyze_only:
        if (outputs / mat["restart"]).exists():
            print(f"equilibrated ({mat['restart']} exists), skipping")
        else:
            run_deck(f"inputs/in.cascade_{args.material}_equilibrate",
                     "equilibrate", mat["equil_steps"], args, cwd)

        natoms = natoms_from_data(structures / mat["reference"])
        for k in range(1, args.runs + 1):
            rng = random.Random(args.seed + k)
            nx, ny, nz = pka_direction(rng)
            lmp_vars = {"seed": args.seed + k,
                        "pkaid": mat.get("pkaid", natoms // 2 + k * 137),
                        "runid": k, "nx": nx, "ny": ny, "nz": nz}
            run_deck(f"inputs/in.cascade_{args.material}_cascade",
                     f"cascade {k}/{args.runs}", mat["cascade_steps"],
                     args, cwd, lmp_vars)

    trajs = []
    for k in range(1, args.runs + 1):
        relax = outputs / mat["traj"].format(k=k)
        spike = outputs / mat["traj"].replace("relax", "spike").format(k=k)
        usable = [t for t in (relax, spike)
                  if t.exists() and t.stat().st_size > 0]
        if usable:
            trajs.append(usable[0])
    if not trajs:
        sys.exit("no non-empty trajectories to analyze")
    py = analysis_python()
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen",
               LD_LIBRARY_PATH=str(Path(py).parents[1] / "lib"))
    subprocess.run(
        [py, str(HERE / "analysis" / "defects.py"),
         *map(str, trajs), "--reference", str(structures / mat["reference"]),
         "--material", args.material,
         "--out", str(outputs / "results" / f"results_{args.material}.csv")],
        env=env, check=True)
    print(f"results: {outputs / 'results' / f'results_{args.material}.csv'}")


if __name__ == "__main__":
    main()
