#!/usr/bin/env python3
"""Threshold displacement energy E_d by bisection over recoil directions.

Per direction: lowest recoil energy leaving a stable Frenkel pair at ~6 ps
(voronoi occupation count). Sublattice-resolved for SiC.

    python3 ed.py SiC --dirs 10 --np 2
"""

import argparse
import csv
import random
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# species: (label, atom type, scan lo..hi, step [eV])
SPECS = {
    "Fe": [("Fe", 1, 10.0, 250.0, 5.0)],
    "W": [("W", 1, 25.0, 350.0, 5.0)],
    "SiC": [("Si", 1, 10.0, 140.0, 5.0), ("C", 2, 5.0, 80.0, 2.5)],
}


def uniform_direction(rng: random.Random):
    """Uniform random unit vector; E_d averaging must include channels."""
    while True:
        v = [rng.gauss(0, 1) for _ in range(3)]
        norm = sum(x * x for x in v) ** 0.5
        if norm > 1e-6:
            return [x / norm for x in v]


def survives(material: str, ktype: int, e0: float, d, tag: str,
             args) -> bool:
    """Run one recoil; True if a vacancy survives."""
    log = args.workdir / "outputs" / "ed" / f"log.ed_{material}_{tag}"
    cmd = ["mpirun", "-np", str(args.np), args.lmp,
           "-in", f"inputs/in.ed_{material}", "-log", str(log),
           "-var", "e0", f"{e0:.2f}", "-var", "ktype", str(ktype),
           "-var", "nx", f"{d[0]:.5f}", "-var", "ny", f"{d[1]:.5f}",
           "-var", "nz", f"{d[2]:.5f}"]
    subprocess.run(cmd, cwd=args.workdir, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    text = log.read_text()
    if "1 atoms in group pka" not in text:
        sys.exit(f"pka group != 1 atom, see {log}")
    m = re.search(r"EDRESULT ktype \d+ e0 \S+ vac (\d+)", text)
    if not m:
        sys.exit(f"no EDRESULT in {log}")
    return int(m.group(1)) >= 1


def e_d_direction(material: str, label: str, ktype: int, lo: float,
                  hi: float, step: float, d, i: int, args) -> float | None:
    """Ramp upward from lo; first defect-producing energy is E_d.

    Defect production is non-monotonic near channels (windows open and
    close), so bisection overshoots; a ramp scan does not. Step doubles
    above 100 eV.
    """
    e, k = lo, 0
    while e <= hi:
        if survives(material, ktype, e, d, f"{label}_{i}_r{k}", args):
            return e
        e += step if e < 100.0 else 2 * step
        k += 1
    print(f"  {label} dir{i}: no defect up to {hi} eV", file=sys.stderr)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Threshold displacement energy by bisection.")
    parser.add_argument("material", choices=sorted(SPECS))
    parser.add_argument("--dirs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--np", type=int, default=2, help="MPI ranks")
    parser.add_argument("--lmp", default="lmp")
    parser.add_argument("--workdir", type=Path, default=HERE / "lammps")
    args = parser.parse_args()

    (args.workdir / "outputs" / "ed").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    directions = [uniform_direction(rng) for _ in range(args.dirs)]

    out = args.workdir / "outputs" / "results" / f"ed_results_{args.material}.csv"
    rows = []
    for label, ktype, lo, hi, step in SPECS[args.material]:
        for i, d in enumerate(directions):
            ed = e_d_direction(args.material, label, ktype, lo, hi, step,
                               d, i, args)
            rows.append({"material": args.material, "species": label,
                         "dir": i, "nx": round(d[0], 5), "ny": round(d[1], 5),
                         "nz": round(d[2], 5),
                         "e_d_eV": round(ed, 2) if ed else ""})
            print(f"{label} dir{i} ({d[0]:.3f} {d[1]:.3f} {d[2]:.3f}): "
                  f"E_d = {ed if ed else 'n/a'} eV")

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")

    for label, *_ in SPECS[args.material]:
        vals = [r["e_d_eV"] for r in rows
                if r["species"] == label and r["e_d_eV"] != ""]
        if vals:
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals)
                   / max(len(vals) - 1, 1)) ** 0.5
            print(f"E_d({label}): mean {mean:.1f} eV  std {std:.1f}  "
                  f"n={len(vals)}")


if __name__ == "__main__":
    main()
