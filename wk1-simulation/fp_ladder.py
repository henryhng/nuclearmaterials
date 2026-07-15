#!/usr/bin/env python3
"""Frenkel-pair accumulation ladder: cascade-statistics defect populations
inserted at rising concentration, relaxed at zero pressure.

Per rung: stored energy, lattice swelling; dumps for ADP/PDF extraction.
Defect mix mirrors the 10 keV cascade debris: 2:1 C:Si Frenkel pairs
(dumbbell-seeded interstitials), antisite pairs at 0.16 per FP.

    python3 fp_ladder.py --np 6
"""

import argparse
import csv
import random
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

CONCS = [0.0, 1e-4, 5e-4, 2e-3, 8e-3, 2e-2, 5e-2]
C_FRACTION = 2 / 3          # of FPs on the carbon sublattice
ANTISITE_PER_FP = 0.16
EV_ATOM_TO_J_G = 96485.33 / 20.05


def read_data(path: Path):
    """Header lines, atom rows [id, type, x, y, z], box edge length."""
    lines = path.read_text().splitlines()
    atoms, header, in_atoms = [], [], False
    box = None
    for line in lines:
        if line.strip().startswith("Atoms"):
            in_atoms = True
            header.append(line)
            continue
        if in_atoms and line.split():
            p = line.split()
            if len(p) >= 5 and p[0].isdigit():
                atoms.append([int(p[0]), int(p[1]),
                              float(p[2]), float(p[3]), float(p[4])])
                continue
            in_atoms = False
        if not atoms:
            header.append(line)
        if "xlo xhi" in line:
            lo, hi = float(line.split()[0]), float(line.split()[1])
            box = hi - lo
    return header, atoms, box


def build(atoms, box, conc, rng):
    """Insert FP + antisite population; returns modified atom rows."""
    n_fp = round(conc * len(atoms))
    if n_fp == 0:
        return atoms
    by_type = {1: [i for i, a in enumerate(atoms) if a[1] == 1],
               2: [i for i, a in enumerate(atoms) if a[1] == 2]}
    n_c = round(C_FRACTION * n_fp)
    picks = (rng.sample(by_type[2], n_c)
             + rng.sample(by_type[1], n_fp - n_c))

    moved = set(picks)
    for i in picks:
        # dumbbell seed: park the atom 1.4 A off a random same-species host
        host = rng.choice(by_type[atoms[i][1]])
        while host in moved:
            host = rng.choice(by_type[atoms[i][1]])
        d = [rng.gauss(0, 1) for _ in range(3)]
        norm = sum(x * x for x in d) ** 0.5
        for k in range(3):
            atoms[i][2 + k] = (atoms[host][2 + k]
                               + 1.4 * d[k] / norm) % box

    n_anti = round(ANTISITE_PER_FP * n_fp / 2)
    for _ in range(n_anti):
        i = rng.choice(by_type[1])
        j = rng.choice(by_type[2])
        if i in moved or j in moved:
            continue
        atoms[i][1], atoms[j][1] = 2, 1
    return atoms


def write_data(path: Path, header, atoms):
    while header and not header[-1].strip():
        header.pop()
    with open(path, "w") as f:
        f.write("\n".join(header) + "\n\n")
        for a in atoms:
            f.write(f"{a[0]} {a[1]} {a[2]:.6f} {a[3]:.6f} {a[4]:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="FP accumulation ladder.")
    parser.add_argument("--np", type=int, default=6)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--lmp", default="lmp")
    parser.add_argument("--workdir", type=Path, default=HERE / "lammps")
    args = parser.parse_args()

    out = args.workdir / "outputs" / "fpladder"
    out.mkdir(parents=True, exist_ok=True)
    header, atoms0, box = read_data(
        args.workdir / "structures" / "perfect_lattice_SiC.data")

    rows = []
    for conc in CONCS:
        tag = f"c{conc:g}".replace("-", "m").replace(".", "p")
        rng = random.Random(args.seed)
        atoms = build([a[:] for a in atoms0], box, conc, rng)
        data = out / f"sic_{tag}.data"
        write_data(data, header, atoms)

        log = out / f"log.fpladder_{tag}"
        subprocess.run(
            ["mpirun", "-np", str(args.np), args.lmp,
             "-in", "inputs/in.fpladder_SiC", "-log", str(log),
             "-var", "datafile", str(data), "-var", "tag", tag],
            cwd=args.workdir, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        m = re.search(r"LADDER \S+ pe (-?[\d.]+) lx ([\d.]+)",
                      log.read_text())
        rows.append({"conc_fp_per_atom": conc, "tag": tag,
                     "pe_eV": float(m.group(1)), "lx_A": float(m.group(2))})
        print(f"{tag}: pe {rows[-1]['pe_eV']:.1f}  lx {rows[-1]['lx_A']:.4f}")

    pe0, lx0 = rows[0]["pe_eV"], rows[0]["lx_A"]
    n = len(atoms0)
    for r in rows:
        r["stored_meV_atom"] = round((r["pe_eV"] - pe0) / n * 1e3, 4)
        r["stored_J_g"] = round((r["pe_eV"] - pe0) / n * EV_ATOM_TO_J_G, 2)
        r["da_over_a"] = round((r["lx_A"] - lx0) / lx0, 6)

    csvpath = out / "ladder_results.csv"
    with open(csvpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csvpath}")


if __name__ == "__main__":
    main()
