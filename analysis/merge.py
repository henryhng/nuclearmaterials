#!/usr/bin/env python3
"""Concatenate per-material results CSVs -> one master CSV.

Material inferred from results_<Mat>.csv filenames; columns are the union.

    python3 merge.py ../lammps/outputs/results_*.csv \
        --out ../lammps/outputs/results_master.csv
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np


def material_label(path: Path) -> str:
    m = re.search(r"results_(\w+)\.csv$", path.name)
    return m.group(1) if m else path.stem


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-material results CSVs into a master CSV.")
    parser.add_argument("csvs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results_master.csv"))
    args = parser.parse_args()

    rows, fields = [], ["material"]
    for path in sorted(args.csvs):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            fields += [c for c in reader.fieldnames if c not in fields]
            for r in reader:
                rows.append({"material": material_label(path), **r})

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.out} ({len(rows)} runs, {len(fields)} columns)")
    for mat in dict.fromkeys(r["material"] for r in rows):
        sub = [r for r in rows if r["material"] == mat]
        fp = [float(r["frenkel_pairs"]) for r in sub if r.get("frenkel_pairs")]
        se = [float(r["eV_per_fp"]) for r in sub if r.get("eV_per_fp")]
        line = f"{mat}: {len(sub)} runs"
        if fp:
            line += f"  FP mean {np.mean(fp):.1f}"
        if se:
            line += f"  eV/FP mean {np.mean(se):.2f}"
        print(line)


if __name__ == "__main__":
    main()
