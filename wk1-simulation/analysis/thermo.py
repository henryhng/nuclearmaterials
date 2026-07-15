#!/usr/bin/env python3
"""Temperature and potential energy vs step from LAMMPS logs.

    python3 thermo.py ../logs/log.* --outdir ../figures/thermo
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


KEEP = ("Step", "Temp", "PotEng")


def parse_log(path: Path) -> dict:
    """Thermo sections concatenated on shared columns: {column: array}."""
    cols, out = None, {c: [] for c in KEEP}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if parts and parts[0] == "Step":
                cols = parts
                continue
            if cols is None or len(parts) != len(cols):
                continue
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue
            for c in KEEP:
                out[c].append(vals[cols.index(c)])
    return {c: np.array(v) for c, v in out.items()}


def plot_log(path: Path, outdir: Path) -> Path:
    d = parse_log(path)
    step, temp, pe = d["Step"], d["Temp"], d["PotEng"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(step, temp, lw=0.8, color="tab:red")
    ax1.set_ylabel("Temperature (K)")
    ax1.axhline(300, color="gray", ls="--", lw=0.7)
    ax1.grid(alpha=0.3)

    ax2.plot(step, pe, lw=0.8, color="tab:blue")
    ax2.set_ylabel("Potential energy (eV)")
    ax2.set_xlabel("Step")
    ax2.grid(alpha=0.3)

    ax1.set_title(path.name)

    fig.tight_layout()
    out = outdir / f"thermo_{path.name.replace('log.', '').replace('run_', '')}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description="Thermo plots from logs.")
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--outdir", type=Path, default=Path("."))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    for log in args.logs:
        print(plot_log(log, args.outdir))


if __name__ == "__main__":
    main()
