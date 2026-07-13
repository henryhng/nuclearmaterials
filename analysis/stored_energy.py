#!/usr/bin/env python3
"""Defect stored energy from cascade logs -> stored_energy.csv.

Pre-PKA vs post-relaxation PotEng difference at matched temperature.
Needs no threshold energy or cluster classification; comparable to
calorimetry (J/g) and per-defect formation energies (eV/FP).

    python3 stored_energy.py ../lammps/outputs/log.run_SiC_cascade_* \
        --material SiC --results results.csv --out stored_energy.csv
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

import materials


KEEP = ("Step", "Temp", "PotEng", "Time")
EV_ATOM_TO_J_G = 96485.33  # (eV/atom) * this / A[g/mol] -> J/g
MERGE_COLS = ("stored_eV", "stored_meV_atom", "stored_J_g", "eV_per_fp",
              "drift_eV_ps", "plateaued")


def run_label(log_path: Path) -> str:
    m = re.search(r"_(\d+)$", log_path.name)
    return m.group(1) if m else log_path.stem


def parse_blocks(path: Path) -> tuple[list[dict], int]:
    """Per-run thermo blocks {column: array} plus the atom count."""
    blocks, cols, cur = [], None, None
    natoms = 0
    with open(path) as f:
        for line in f:
            m = re.search(r"with (\d+) atoms", line)
            if m:
                natoms = int(m.group(1))
            parts = line.split()
            if parts and parts[0] == "Step":
                cols = parts
                cur = {c: [] for c in KEEP if c in cols}
                blocks.append(cur)
                continue
            if cols is None or len(parts) != len(cols):
                continue
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue
            for c in cur:
                cur[c].append(vals[cols.index(c)])
    return ([{c: np.array(v) for c, v in b.items()} for b in blocks if
             b["Step"]], natoms)


def tail_stats(block: dict, window: int) -> dict:
    """Mean PE/Temp and linear PE drift over the last `window` samples."""
    n = min(window, len(block["PotEng"]))
    pe = block["PotEng"][-n:]
    t = (block["Time"][-n:] if "Time" in block
         else block["Step"][-n:] / 1000.0)  # fallback: kilosteps
    drift = np.polyfit(t, pe, 1)[0] if n > 1 else 0.0
    noise = float(np.std(pe - np.polyval(np.polyfit(t, pe, 1), t))
                  ) if n > 1 else 0.0
    return {"pe": float(pe.mean()), "temp": float(block["Temp"][-n:].mean()),
            "drift": float(drift), "noise": noise,
            "span": float(t[-1] - t[0]) if n > 1 else 0.0}


def analyze(log: Path, mat: dict | None, window: int,
            frenkel: dict | float | None) -> dict | None:
    blocks, natoms = parse_blocks(log)
    if len(blocks) < 2 or not natoms:
        print(f"skip {log.name}: need equilibration + cascade blocks",
              file=sys.stderr)
        return None

    pre, post = tail_stats(blocks[0], window), tail_stats(blocks[-1], window)
    de = post["pe"] - pre["pe"]
    de_atom = de / natoms

    # Plateaued: projected drift over the window within the noise band.
    plateaued = abs(post["drift"] * post["span"]) <= 2 * max(post["noise"],
                                                             1e-12)
    row = {
        "run": run_label(log),
        "log": log.name,
        "n_atoms": natoms,
        "pe_pre_eV": round(pre["pe"], 1),
        "pe_post_eV": round(post["pe"], 1),
        "stored_eV": round(de, 1),
        "stored_meV_atom": round(de_atom * 1e3, 4),
        "temp_pre_K": round(pre["temp"], 1),
        "temp_post_K": round(post["temp"], 1),
        "drift_eV_ps": round(post["drift"], 1),
        "plateaued": plateaued,
    }
    if mat:
        row["stored_J_g"] = round(de_atom * EV_ATOM_TO_J_G / mat["A"], 2)

    fp = frenkel.get(row["run"]) if isinstance(frenkel, dict) else frenkel
    if fp:
        row["frenkel_pairs"] = int(fp)
        row["eV_per_fp"] = round(de / fp, 2)

    if abs(pre["temp"] - post["temp"]) > 5.0:
        print(f"warn {log.name}: window temps differ "
              f"({pre['temp']:.0f} vs {post['temp']:.0f} K), thermal PE "
              f"does not cancel", file=sys.stderr)
    if not plateaued:
        print(f"warn {log.name}: PE still drifting {post['drift']:+.1f} "
              f"eV/ps at end of relaxation, stored energy overestimated",
              file=sys.stderr)
    return row


def merge_into(results: Path, rows: list) -> None:
    """Join stored-energy columns into a defects.py results CSV on run."""
    by_run = {r["run"]: r for r in rows}
    with open(results, newline="") as f:
        reader = csv.DictReader(f)
        data = list(reader)
        fields = list(reader.fieldnames)
    fields += [c for c in MERGE_COLS if c not in fields]
    for row in data:
        src = by_run.get(row["run"], {})
        row.update({c: src.get(c, row.get(c, "")) for c in MERGE_COLS})
    with open(results, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def load_frenkel(results: Path) -> dict:
    """{run: frenkel_pairs} from a defects.py results.csv."""
    with open(results, newline="") as f:
        return {r["run"]: float(r["frenkel_pairs"])
                for r in csv.DictReader(f) if r.get("frenkel_pairs")}


def main():
    parser = argparse.ArgumentParser(
        description="Defect stored energy from cascade logs -> CSV.")
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--material", choices=sorted(materials.MATERIALS),
                        help="enable J/g via molar mass")
    parser.add_argument("--results", type=Path,
                        help="defects.py results.csv for eV per Frenkel pair")
    parser.add_argument("--frenkel", type=float,
                        help="Frenkel pair count (overrides --results)")
    parser.add_argument("--window", type=int, default=50,
                        help="thermo samples averaged per plateau")
    parser.add_argument("--merge", action="store_true",
                        help="write stored columns into --results instead")
    parser.add_argument("--out", type=Path, default=Path("stored_energy.csv"))
    args = parser.parse_args()
    if args.merge and not args.results:
        parser.error("--merge requires --results")

    mat = materials.get(args.material) if args.material else None
    frenkel = args.frenkel or (load_frenkel(args.results) if args.results
                               else None)
    rows = [r for log in sorted(args.logs)
            if (r := analyze(log, mat, args.window, frenkel))]
    if not rows:
        sys.exit("no usable logs")

    if args.merge:
        merge_into(args.results, rows)
        print(f"Merged into {args.results} ({len(rows)} runs)")
    else:
        fields = list(rows[0].keys())
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.out} ({len(rows)} runs)")
    de = np.array([r["stored_eV"] for r in rows])
    print(f"Stored energy: mean {de.mean():.0f} eV  "
          f"({np.mean([r['stored_meV_atom'] for r in rows]):.3f} meV/atom)")
    if mat:
        print(f"Calorimetric: mean "
              f"{np.mean([r['stored_J_g'] for r in rows]):.2f} J/g")
    if any("eV_per_fp" in r for r in rows):
        efp = [r["eV_per_fp"] for r in rows if "eV_per_fp" in r]
        print(f"Per Frenkel pair: mean {np.mean(efp):.2f} eV/FP")


if __name__ == "__main__":
    main()
